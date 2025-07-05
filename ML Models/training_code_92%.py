# Training Function with 80/20 Split

def train_rf_fingerprinter_final():
    print("🚀 RF FINGERPRINTING - FINAL TRAINING WITH DISTANCE LABELS")
    print("=" * 70)
    
    # Pre-Train Test
    print("✈️ PRE-FLIGHT CHECKLIST:")
    print(f"✅ Dataset loaded: {len(df):,} samples")
    print(f"✅ Unique classes: {df['label'].nunique()}")
    print(f"✅ Device: {device}")
    
    # Verify we have distance labels
    sample_labels = df['label'].iloc[:5].tolist()
    has_distance = all('_' in label for label in sample_labels)
    if has_distance:
        print("✅ Distance labels confirmed")
    else:
        print("❌ WARNING: Distance labels not found!")
        return None
    
    # Create dataset
    dataset = OracleDataset(df)
    num_classes = len(dataset.labels)
    print(f"✅ Classes to learn: {num_classes}")
    
    # check for classes
    if num_classes != 176:
        print(f"⚠️ WARNING: Expected 176 classes, got {num_classes}")
    
    # 80/20 split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    print(f"✅ Training samples: {len(train_dataset):,}")
    print(f"✅ Testing samples: {len(test_dataset):,}")
    
    # Data loaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Make sure we have the correct # classes here
    model = RFFingerprinter(176)  # num_classes
    model = model.to(device)
    
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✅ Output classes: {model.fc_out.out_features}")
    
    # Verify model matches dataset
    if model.fc_out.out_features != num_classes:
        print(f"❌ CRITICAL ERROR: Model outputs {model.fc_out.out_features} classes but dataset has {num_classes}")
        return None
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4) # Huge addition
    
    # Training parameters
    num_epochs = 200
    best_train_acc = 0.0
    patience = 25
    patience_counter = 0
    
    train_losses = []
    train_accuracies = []
    
    print(f"\n🏋️ TRAINING START - {num_epochs} epochs")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        start_time = time.time()
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Move to device
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
        
        # Epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct_predictions / total_predictions
        epoch_time = time.time() - start_time
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # Early stopping
        if epoch_accuracy > best_train_acc:
            best_train_acc = epoch_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_distance_labeled.pth')
        else:
            patience_counter += 1
        
        # Progress output
        if epoch < 10 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Loss: {epoch_loss:.4f} | "
                  f"Acc: {epoch_accuracy:.2f}% | "
                  f"Best: {best_train_acc:.2f}% | "
                  f"Time: {epoch_time:.1f}s")
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print("\n" + "=" * 70)
    print("🏁 TRAINING COMPLETED - EVALUATING RESULTS")
    
    # Load best model
    model.load_state_dict(torch.load('best_model_distance_labeled.pth'))
    
    # Test evaluation
    model.eval()
    test_predictions = []
    test_targets = []
    test_loss = 0.0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            test_predictions.extend(predicted.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
    
    test_loss = test_loss / len(test_loader)
    combo_accuracy = accuracy_score(test_targets, test_predictions)
    
    print(f"📊 COMBINATION ACCURACY (Transmitter + Distance):")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Accuracy: {combo_accuracy:.4f} ({combo_accuracy*100:.2f}%)")
    
    # TRANSMITTER-ONLY ANALYSIS
    print(f"\n📡 TRANSMITTER-ONLY ACCURACY:")
    tx_accuracy, tx_accuracies = get_transmitter_accuracy(test_predictions, test_targets, dataset.labels)
    print(f"   Overall Transmitter Accuracy: {tx_accuracy:.4f} ({tx_accuracy*100:.2f}%)")
    
    print(f"\n📈 Per-Transmitter Accuracy:")
    sorted_tx = sorted(tx_accuracies.items(), key=lambda x: x[1], reverse=True)
    for tx_id, acc in sorted_tx:
        print(f"   {tx_id}: {acc:.2%}")
    
    # Best and worst transmitters
    best_tx = max(tx_accuracies.items(), key=lambda x: x[1])
    worst_tx = min(tx_accuracies.items(), key=lambda x: x[1])
    
    print(f"\n🏆 Best transmitter: {best_tx[0]} ({best_tx[1]:.2%})")
    print(f"😓 Worst transmitter: {worst_tx[0]} ({worst_tx[1]:.2%})")
    
    # Distance-combo analysis (top few)
    print(f"\n🎯 Best Transmitter-Distance Combinations:")
    cm = confusion_matrix(test_targets, test_predictions)
    combo_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    # Get top 10 combinations
    top_combos = sorted(zip(dataset.labels, combo_accuracies), key=lambda x: x[1], reverse=True)
    for i, (combo_label, acc) in enumerate(top_combos[:10]):
        print(f"   {i+1:2d}. {combo_label}: {acc:.2%}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies)
    plt.title('Training Accuracy (Combo)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    # Transmitter accuracy histogram
    tx_acc_values = list(tx_accuracies.values())
    plt.hist(tx_acc_values, bins=10, alpha=0.7)
    plt.title('Transmitter Accuracy Distribution')
    plt.xlabel('Accuracy')
    plt.ylabel('Count')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Final summary
    print(f"\n🎯 FINAL SUMMARY:")
    print("=" * 50)
    print(f"Dataset: {len(dataset):,} samples, {num_classes} classes")
    print(f"Combination Accuracy: {combo_accuracy:.2%}")
    print(f"Transmitter Accuracy: {tx_accuracy:.2%}")
    print(f"Best Transmitter: {best_tx[0]} ({best_tx[1]:.2%})")
    print(f"Worst Transmitter: {worst_tx[0]} ({worst_tx[1]:.2%})")
    
    
    # Save final model
    torch.save(model.state_dict(), 'rf_fingerprinter_distance_final.pth')
    
    return model, dataset, tx_accuracy

model, dataset, tx_accuracy = train_rf_fingerprinter_final()